# Experiments after Stage B: results and code guide

This guide separates scientific experiments from software tests. Stage B tested whether an eight-dimensional, tree-only GNN representation helps Bayesian optimization (BO). The subsequent experiments investigated neighborhood search, the physical source of compilation cost, longer GNN training, and finally screening without circuit-cost labels. Stage C's larger corpus and mixed random–SCENT pretraining remain on hold.

## 1. What has been tested?

| Experiment | Data and budget | Main observation | Interpretation |
|---|---|---|---|
| Additional Stage B diagnostics | Existing six targets; 128 candidates each; examine smaller budgets and good-candidate density | Random search remains competitive; low-cost trees are common in overlapping targets 0 and 2 but rare in target 1 | Pool size alone does not explain GNN performance; target-specific difficulty matters |
| Independent neighborhood validation | Ten new eight-qubit targets, seeds 3–7; three search seeds; SCENT plus 8/16/32 additional evaluations | Initial-neighborhood search loses to random on all five overlapping targets; iterative search wins on one and loses on four | The earlier neighborhood advantage does not generalize consistently |
| Local compilation mechanism audit | Accepted and unsuccessful moves; detailed overlapping_4 trajectory | A 504→279 CNOT transition consists of two block reductions, 129 and 96 gates | Register padding provides a concrete cost mechanism, although partitions and tensor entries also change |
| Truncation sensitivity | The before/after trees at five tolerance multipliers, 0.8–1.2 | The same 504→279 reduction survives all five settings | The observed transition is stable over this local tolerance interval |
| Descriptor retrieval | 2,824 existing evaluations; primary analysis uses 16 fixed cohorts, excluding adaptively chosen candidates | Padded-size score has overlapping-target rank correlations 0.971–0.988 and very small top-five regret | A target-dependent, label-free screening rule merits independent evaluation |
| Extended GNN training | Same 2,000 training trees and architecture; all three seeds; 100→300 epochs | Reconstruction improves strongly; all three seeds improve the 64-evaluation endpoint, but gains remain inconsistent at smaller budgets | Longer training helped, but it does not establish a reliable advantage over simpler selection rules |
| Fresh screening benchmark | Ten untouched targets, seeds 8–12; identical 128-tree pools; three repetitions; K=1/2/4/8/16 | Padded screening improves early selection and two-second costs; parameter screening catches up by K=16 | Supports screening on these instances under the reference compiler |

Reports:

- [Stage B and its added diagnostics](artifacts/stage_b/REPORT.md)
- [Neighborhood validation](artifacts/neighborhood_validation/REPORT.md)
- [Bounded decision experiment](artifacts/decision_experiment/REPORT.md)
- [Fresh screening benchmark](artifacts/screening_benchmark/REPORT.md)

These experiments use dense, eight-qubit states and our reference isometry compiler. The SCENT adapter supplies a tree, not the paper's full approximate compilation pipeline.

## 2. Important corrections to the original Stage B interpretation

### High fidelity does not mean every tree has the same cost

There is no explicit maximum bond dimension, fixed CNOT cap, or enforced time/memory limit in the evaluator. The SVD keeps as many singular values as the representation tolerance requires. The default final fidelity is 0.999; representation infidelity is allocated 0.0001 and compilation infidelity 0.0004. Precision retries multiply the initial local truncation tolerance by 1, 0.1, and 0. The zero-tolerance case still removes numerical zeros. Evaluation stops at the first feasible attempt.

Consequently, a poor topology can reach the same fidelity by using larger tensors and more gates. In the original hierarchical pools, CNOT counts span 75–777. Search curves all staying at 75 mean the common initialization already contains the 75-CNOT SCENT tree; they do not mean all trees are equivalent.

### Random search is a sequence of evaluations

It samples unseen trees and keeps the best verified result so far. Original Stage B methods share eight initial evaluations, including SCENT. Thus random search can improve SCENT after spending additional computation; this is not evidence that it is cheaper than constructing SCENT once.

| Original overlapping target | Candidates | Best CNOT | Within 5% of best | Within 10% |
|---|---:|---:|---:|---:|
| 0 | 128 | 354 | 11 | 28 |
| 1 | 128 | 277 | 1 | 1 |
| 2 | 128 | 356 | 11 | 28 |

After the shared eight evaluations, the mean best overlapping CNOT was:

| Method | 4 extra | 8 extra | 16 extra |
|---|---:|---:|---:|
| Random | 352.89 | 349.33 | 348.00 |
| Neighborhood-first | 349.56 | 340.33 | 336.78 |
| Raw structural-feature BO | 356.11 | 352.78 | 342.89 |

The original GNN also lacks an advantage at these small budgets. A large budget-to-pool ratio explains some random-search strength, but not the entire result.

## 3. Independent neighborhood validation

### Experimental results

`configs/neighborhood_validation.json` freezes new target seeds 3–7 in each family. Each run starts from SCENT and receives 8, 16, or 32 additional candidate evaluations. Three random seeds produce 90 trajectories and 2,880 search steps, with 2,056 unique evaluations after cache sharing.

| Method | Overlapping, +8 | +16 | +32 |
|---|---:|---:|---:|
| Global random | 364.73 | 354.40 | 354.13 |
| Initial neighborhood | 503.47 | 380.27 | 360.93 |
| Iterative neighborhood | 484.60 | 452.47 | 344.73 |

The final mean favors iterative search, but the per-target evidence is weak: it wins on only overlapping_4 and loses on the other four. All hierarchical results remain at SCENT's 75 CNOTs. This is why we did not expand neighborhood search immediately.

### How the code works

`scripts/run_neighborhood_validation.py` contains `search(start_tree, oracle, method, seed, budget, n)`:

1. Evaluate the starting SCENT tree and initialize the incumbent.
2. Maintain a set of tree hashes already evaluated in that run.
3. Choose the next unseen candidate according to the method.
4. Call the oracle, which constructs, compiles, and verifies the candidate through the existing evaluator.
5. Replace the incumbent only with a feasible tree having strictly fewer CNOTs.
6. Record the candidate, its origin, the incumbent, and evaluation accounting.

The three policies differ in candidate generation:

- **Global random:** generate a new tree by random pair merging, rejecting duplicates. This distribution is not uniform over all labeled binary-tree topologies.
- **Initial neighborhood:** randomly order the unseen one-step neighbors of the original SCENT tree; use global random candidates when that set is exhausted.
- **Iterative neighborhood:** sample unseen neighbors of the current incumbent; a strict improvement changes the center and therefore its neighborhood. Fall back to global random when no unseen neighbor remains. An improving fallback candidate can also become the new center.

`trees.neighbors` uses local rooted-tree rotations. For example, a local grouping `((A,B),C)` can become `((A,C),B)` or `((B,C),A)`. The subtrees retain their physical leaf labels. This changes a partition; it is not an arbitrary qubit relabeling. Canonicalization removes left/right ordering duplicates.

This differs from the original Stage B `neighborhood-first`, which prioritizes neighbors already present in a fixed pool. Dynamic neighborhood validation can generate new trees after observing improvements.

`scripts/analyze_neighborhood_validation.py` aggregates traces, separates target-level results, and examines representation and local circuit changes. Cached evaluations make this an efficient scientific comparison, but the resulting charged times are not independent, fresh wall-time measurements. The new screening benchmark removes that limitation.

### Mechanism inspection and its limits

The audit examines both improving and non-improving moves. The especially successful overlapping_4 path was inspected after observing results, so it is an exploratory case study, not independent confirmation of a general rule.

The path contains 707→504→279→273 CNOTs. At 504→279, maximum rank drops from 9 to 8 and its register width from four qubits to three. Total parameter count changes from 400 to 368. Two affected compiled blocks explain the entire 225-gate reduction:

| Block | Padded matrix before | After | CNOT before | After |
|---|---|---|---:|---:|
| Six-leaf mapping | 64 × 4 | 32 × 4 | 251 | 122 |
| Changed four-leaf mapping | 16 × 16 | 16 × 8 | 218 | 122 |

The remaining blocks contribute 35 gates in both circuits. A rank crossing a power of two changes register allocation discretely; the adjacent block's output width also changes. However, the partitions and actual tensor entries change too. This is not a controlled intervention on rank alone. Separately synthesized local block counts need not sum to a globally optimized circuit in every example; they happen to account for this transition.

## 4. The bounded decision experiment

### File responsibilities

| File | Responsibility |
|---|---|
| `configs/decision_experiment.json` | Fixed transition, tolerance sweep, cohort settings, training length and seeds |
| `scripts/decision_common.py` | Freeze provenance, construct initial representations, traverse nodes, compute descriptors |
| `scripts/decision_cost.py` | Compile transition variants, extract descriptors, evaluate held-out retrieval and replay fixed screening |
| `scripts/decision_gnn.py` | Retrain the unchanged encoder for longer and reevaluate retrieval/search |
| `scripts/decision_report.py` | Aggregate results without dropping failed or undefined metrics silently |
| `scripts/audit_decision_experiment.py` | Check records, model provenance, trajectories and accounting |

### Descriptors: what is computed, and why?

`construct` builds a representation at the **initial, predeclared truncation tolerance**. It does not read the ranks of whichever retry eventually succeeded. Using the successful retry's ranks as a predictor would give the predictor information that was unavailable before evaluation.

`nodes` traverses the representation. `descriptors` collects eight numbers:

1. Maximum bond rank.
2. Sum of bond ranks.
3. Representation parameter count.
4. Sum of internal-map input-register widths.
5. Sum of internal-map output-register widths.
6. Maximum output-register width.
7. Sum of padded matrix-entry counts.
8. Maximum padded matrix-entry count.

For an internal node with parent-facing rank `r` and child ranks `r1`, `r2`, the reference compiler uses

```text
input_bits  = ceil(log2(r))
output_bits = ceil(log2(r1)) + ceil(log2(r2))
padded_entries = 2 ** (input_bits + output_bits)
```

Rank one needs zero register bits. Each child rank is padded separately, matching the compiler's child registers. Padding the product of the child ranks as a single register would not always give the same size.

The fixed score is the sum of `padded_entries` over internal isometries. Lower is preferred. It is target-dependent because the SVD ranks depend on the target state, not just the tree. It needs no fitted coefficients or circuit-cost labels. Parameter-count screening is a separate fixed baseline using the unpadded representation's stored parameter count.

These descriptors are inexpensive relative to compilation here, but not free topology features: each requires constructing the target-specific TTN.

### Tolerance experiment

`transition` reconstructs and directly compiles both trees at five multipliers: 0.8, 0.9, 1.0, 1.1, 1.2. These ten evaluations retain the original global fidelity requirements; the multiplier changes the local truncation tolerance. `local_blocks` reports individual padded mappings and compiled block costs, alongside singular-value information.

All five settings retain the 504→279 reduction and feasible final fidelities, approximately 0.999997 and 0.999995. This establishes local stability over the tested interval, not insensitivity to every possible error allocation.

### Retrieval experiment

The script extracts 2,824 old evaluations. Primary selection metrics use fixed cohorts: six original pools of 128 and ten new cohorts containing SCENT plus 96 global random trees. Adaptively generated neighborhood candidates are excluded from this primary comparison to reduce selection bias.

For each cohort, known-label splits of size 8 or 16 support comparisons with learned GP predictors. The fixed scores ignore those known labels: they simply rank the remaining candidates. Therefore, “eight known labels” describes a comparison split, not a fitting requirement for padded-size screening.

The metrics include rank correlation and top-five retrieval regret: how much worse the best retrieved candidate is than the best available candidate in the held-out set. The experiment also examines ties and counterexamples, rather than just successful moves. GP features are standardized using the known subset; costs are modeled on the transformed scale used by the search code. Split seeds are fixed and score ties are resolved independently of cost labels.

There are 800 predictor/split comparisons, including 320 GP fits. Fixed scorers do not fit models.

| Eight-known-label comparison | Padded mean top-five regret | Random mean regret |
|---|---:|---:|
| Original overlapping cohorts | 0.13 | 59.58 |
| New random-sample overlapping cohorts | 0.40 | 41.25 |

Padded-score correlations range from 0.971 to 0.988 across overlapping targets. Parameter-count screening is weaker in held-out retrieval, but it matches padded screening's final mean CNOT in the new random-sample cohorts. Both must remain in the next experiment.

The replay curves start from SCENT and use existing labels. They suggest selection efficiency, but do not establish end-to-end speedup. Rebuilding all 2,824 representations took about 8.29 seconds, or 2.94 ms each on average. That alone is not a timing comparison with random search.

## 5. Controlled longer GNN training

The GNN itself is unchanged. `src/ttn_search/learning/encoder.py` represents each tree as a graph with leaf identity, node type, depth and subtree-size information. Direction-aware message passing distinguishes parents and children. Root, mean and max pooling feed an eight-dimensional embedding. A decoder predicts pairwise LCA depths and leaf depths. The loss encourages structural reconstruction, not low CNOT counts.

For eight leaves, `graph(tree)` produces 15 nodes and 13 features per node: three node-type indicators, eight physical-leaf identity indicators, normalized depth and normalized subtree size. `children[i,j]=1` means that node j is a child of i. `batch` stacks these into tensors of shape `[batch,15,13]` and `[batch,15,15]`.

The encoder first maps 13 features to 64 hidden channels. Each of four layers computes a self contribution, a sum over children, and a sum over parents, with different learned linear maps for the two directions. An MLP transforms their sum. Child order is irrelevant to summation, while physical leaf identities remain visible. Concatenating root, mean and maximum node embeddings gives 192 channels; the readout maps these to the eight-dimensional representation.

The decoder maps eight dimensions to 36 outputs: 28 pairwise LCA depths plus eight leaf depths. `loss` averages the two groups' mean-squared errors with equal weight, so the larger LCA group does not dominate solely by its size. `embed` switches to evaluation mode and disables gradient tracking. The decoder is needed for pretraining, not for selecting candidates downstream.

The encoder sees no target-state amplitudes or bond ranks. Two uses of the same tree on different states receive the same embedding. The BO surrogate can learn different costs from each target's observations, but the embedding itself cannot adapt its description to target-dependent compression. This is an important distinction from the rank/width descriptors.

`scripts/decision_gnn.py` keeps the original 2,000 training trees, validation/test split, architecture, learning rate and three seeds. It reruns training from the beginning for 300 epochs because the original saved checkpoints did not contain optimizer state. The first 100 epochs reproduce the earlier histories exactly; this checks that the extension did not silently change the experiment.

Each model checkpoint is selected by validation reconstruction loss. No downstream target cost selects the checkpoint. The frozen embedding then serves as BO input, while a target-specific GP learns from the small set of evaluated costs.

| Training seed | Test reconstruction loss, 100 epochs | 300 epochs | Overlapping CNOT at budget 64, old | Extended |
|---|---:|---:|---:|---:|
| 0 | 0.009351 | 0.003758 | 345.00 | 331.44 |
| 1 | 0.008552 | 0.003964 | 332.56 | 331.22 |
| 2 | 0.010276 | 0.004615 | 348.00 | 343.11 |

Longer training genuinely improves the final search endpoint for all three seeds. It is incorrect to summarize this as “only reconstruction improved.” However, small-budget effects and retrieval improvements are inconsistent; seed 2 worsens at the smallest reported budget. The extension does not establish a stable advantage over fixed screening or the strongest development baseline.

The extension evaluates 54 search trajectories, 3,456 steps and 180 prediction fits. One constant GP prediction yields an undefined rank correlation; it is recorded as null rather than converted into a favorable value or silently omitted from retrieval accounting.

## 6. Fresh end-to-end screening benchmark

### Frozen protocol

`configs/screening_benchmark.json` specifies five untouched seeds, 8–12, for each family; 128 candidates; three repetitions; and checkpoints at 1, 2, 4, 8 and 16 evaluations. Each pool contains SCENT plus 127 unique random pair-merge trees. It contains no separately inserted balanced, comb or neighborhood prefix.

Six methods are evaluated:

- SCENT alone, one evaluation.
- Uniform random order without replacement.
- SCENT first, then uniform random order: a stronger random baseline.
- Parameter-count screening.
- Padded-size screening.
- Frozen extended GNN plus existing BO, with model seeds 0/1/2 assigned to repetitions 0/1/2.

The two fixed screening methods have **no forced SCENT compilation** and no eight-label initialization. They may rank SCENT first, but must choose it through their score. The GNN starts from SCENT and uses the existing proposal routine; with too few observations it falls back to random selection before fitting its surrogate. All three models are retained rather than choosing the most successful development seed.

### Separating construction from compilation

`src/ttn_search/screening/evaluator.py` introduces three main operations:

**`prepare(target, tree, pipeline, factor)`** constructs the dense SVD representation and checks representation fidelity. It returns a `Prepared` object containing the representation, a provenance identity, and a timed attempt record. It never synthesizes a quantum circuit.

**`scores(prepared)`** traverses internal nodes and returns parameter count and padded-size sum. If construction produced no representation, its scores are infinite. No measured CNOT appears in this function.

**`evaluate_fresh(..., prepared=None)`** performs the original retry schedule and verifies the resulting circuit. When screening supplies a matching `Prepared` object, the first attempt uses that exact representation. If a tighter tolerance is needed, a new representation is built for that retry. Unscreened methods construct only their selected candidates.

The identity hashes the target, tree and pipeline. This prevents accidentally supplying a TTN constructed for another candidate or precision policy. The hash is an integrity/reuse check, not a learned feature.

For every compiled attempt, the evaluator stores the QPY circuit, state arrays, timing, fidelity and error metadata. There is deliberately **no result cache**. Different methods and repetitions synthesize independently, even when they select the same tree.

### Timing without double counting

The first representation's build time stays in its attempt record for provenance. A `reused_initial` flag excludes it from incremental build accounting, because screening already paid that time during whole-pool construction. Retries add their new build costs normally.

`run_one` in `scripts/run_screening_benchmark.py` follows this sequence:

```text
start actual clock
construct SCENT and generate the common candidate pool
if screening:
    prepare every TTN once
    calculate fixed scores and sort with seeded tie breaking
if GNN:
    load frozen model/scaler and embed the candidate pool
for each permitted candidate:
    select the next candidate
    compile and verify, reusing its prepared TTN when available
    update the best feasible cost
    record actual elapsed time and cumulative compiler calls
```

The clock includes pool generation, full screening, model loading, proposals, compilation, retries, verification and artifact writes before a result becomes available. Screening cannot receive a cheap first result while hiding the cost of constructing the other 127 candidates.

State generation is common input preparation and reported separately. Imports, an old-target warmup and exporting scalers from the old training corpus occur outside the online method clock. Each actual method run still regenerates its own pool inside its clock. Prior GNN training is an offline cost. Method order is randomized before seeing labels; execution is sequential with one BLAS/PyTorch thread.

Retry time overlaps build/compile/verify phases. It must not be added to their sum again. Wall time is authoritative; phase timers explain it but omit some allocation and serialization overhead.

### Three budget views

`scripts/report_screening_benchmark.py` uses `at_budget` to return only an incumbent available within a specified budget:

1. **Candidate budget:** how many trees have been evaluated, including failures.
2. **Actual synthesis-call budget:** every attempted compiler call counts, including failed calls and retries. A candidate that finishes beyond the call budget cannot contribute its later result.
3. **Elapsed-time budget:** only completed, verified results with timestamps within the deadline count. There is no interpolation using future results.

A time-budget cell shows both the conditional mean cost and how many runs have a feasible result. A low conditional mean from a few completed runs is not evidence of superiority over a method that finished in every run. The time plot displays a method only where all its runs have an incumbent.

Each search is capped at 16 candidate evaluations. A finished run carries its incumbent forward; it does not use remaining time for additional candidates. SCENT similarly carries its one result. Thus the time comparison is a bounded shortlist benchmark, not an unrestricted anytime-search study.

### Reproducibility and outputs

`main` freezes config, source fingerprints, software, model hashes and corpus hash in `protocol.json` before evaluation. A changed protocol rejects reuse of the output directory. Completed runs can be resumed; an incomplete method run is executed afresh. Shared historical circuit caches are never consulted.

During this run, a concurrent edit expanded the `reference.py` docstring. The original core files were recovered byte-for-byte against the frozen combined hash, and an AST comparison confirmed unchanged executable code after removing docstrings. The running process retained its imported implementation. This is recorded explicitly in `source_review.json`; the source-hash protection was not disabled. A subsequent resume with changed source bytes will still reject the directory.

The benchmark saves each pool, per-method selection trace, per-candidate circuit/attempt record, aggregate runs, timing totals, a Markdown report and a standalone plot. `scripts/audit_screening_benchmark.py` independently reconstructs fixed/random orderings and checks provenance, pool identity, fidelity, retries, reuse flags, cumulative timings and best-so-far curves.

The pool optimum is not known, because only shortlisted candidates are compiled. Claims concern verified selected costs and measured computation, not regret to an exhaustively established optimum.

## 7. Software validation

The full suite currently has 28 passing tests. The relevant additions are:

| Test file | What it protects |
|---|---|
| `tests/test_neighborhood_validation.py` | Neighborhood-search policy, candidate accounting and controlled behavior |
| `tests/test_decision_descriptors.py` | Descriptor dimensions and consistency with representation/compiler conventions |
| `tests/test_screening_evaluator.py` | Real-circuit parity with the original evaluator; initial TTN reuse; forced compiler failures and all retries; padded score dimensions; rejection of mismatched prepared trees |
| `tests/test_screening_budgets.py` | No future result credited at a time or compiler-call boundary; completed SCENT incumbent persists |

Tests verify implementation contracts. The experiment audits validate saved runs and provenance. Neither substitutes for independent targets, actual timing or eventual validation with a different compiler.

## 8. Current decision

Keep Stage C's larger and mixed pretraining on hold. Use the fresh screening report to decide whether fixed target-dependent screening produces useful circuits at fewer compilations and at comparable elapsed time. Keep parameter count as a serious competitor. If the evidence survives, a modest increase in system size or a materially different compiler is the next generalization test; neither is established by the eight-qubit results.

## 9. Completed screening results

All 180 method runs completed: 2,430 candidate evaluations, 2,495 synthesis calls,
65 retry attempts, and approximately 19.9 minutes total execution. Every evaluated
candidate ultimately passed; minimum final fidelity was 0.999892416027. The audit
passed all provenance, ordering, reuse, fidelity and timing checks.

Mean best verified CNOT on the five untouched overlapping targets (three repetitions each):

| Method | K=1 | K=4 | K=16 | At 2 seconds | At 4 seconds |
|---|---:|---:|---:|---:|---:|
| SCENT alone | 700.60 | 700.60 | 700.60 | 700.60 | 700.60 |
| Random | 549.60 | 383.73 | 364.60 | 412.07 | 374.00 |
| SCENT + random | 700.60 | 390.67 | 359.27 | 480.53 | 375.13 |
| Parameter count | 456.20 | 372.93 | 339.80 | 428.00 | 353.87 |
| Padded size | 352.00 | 345.93 | 339.80 | 347.33 | 339.93 |
| Frozen extended GNN + BO | 700.60 | 424.53 | 362.27 | 495.67 | 398.13 |

Every run has a feasible result at the two- and four-second checkpoints. At shorter
deadlines, availability differs, so the full report includes completion fractions.
The actual compiler-call budget gives the same overlapping checkpoints as the
candidate-budget table; retries occur elsewhere or do not change these incumbents.

Against parameter count at K=1, padded screening is better on three overlapping
targets and ties on two. At K=4 it is better on two and ties on three. Both methods
reach the same per-target endpoints by K=16. At two seconds, padded screening is
better on three targets and ties on two. Its early advantage therefore extends
beyond a single unusually successful target, while parameter count remains a
strong final-shortlist baseline.

For the new hierarchical instances, SCENT already averages 75.8 CNOTs. Both
screeners reach 75.2 by K=2, while paying for the entire pool first. At 0.5 seconds,
SCENT has a result in every hierarchical run and neither screener has one. There
is little practical reason here to spend sixteen evaluations chasing a marginal
improvement unless that small gate reduction is itself valuable.

These results support fixed screening as the current baseline for this compiler.
They do not establish a global pool optimum, transfer to larger systems, or
transfer to SCENT's approximate compiler. The next useful generalization test is
a modestly larger system or different compilation backend; larger/mixed GNN
pretraining remains on hold.

## 10. Revised precision policy and resumed bounded GNN comparison

The later development probe changes local truncation tolerance while retaining
all global accuracy requirements. `scripts/revised_policy.py` defines the shared
attempt sequence 3/1/0.1/0, stopping at the first verified feasible circuit. It
wraps synthesis with a flushed start log so retries can be audited. It delegates
construction, fidelity checks and error handling to the existing evaluator.
Representation error must remain ≤0.0001, compilation error ≤0.0004, and final
fidelity ≥0.999. First-3× representations are reused by screening; costs for
rejected attempts remain charged.

`scripts/validate_revised_policy.py` first compares direct 1× and 3× compilation
on six development instances (both families, eight and ten qubits), using three
fixed trees per instance. It then rebuilds the original 128-tree overlapping pool
under the complete policy. Descriptors use the first 3× build, never the accepted
retry's hindsight information. The [policy report](artifacts/revised_policy/REPORT.md)
records the failure that required 1× fallback and the new 592-CNOT/436-depth
SCENT frontier point.

The previously open-ended training hold is superseded only for the authorized
bounded random-data ablation in `scripts/run_revised_gnn.py`; mixed pretraining
is still deferred. `Encoder(n=10)` uses 19 nodes, 15 node features and 55 decoder
outputs. New ten-qubit models are trained; eight-qubit weights are not transferred.
The 2000-tree corpus is a subset of the 10000-tree corpus, and both use common,
disjoint 500-tree validation/test sets. All target-pool topologies are excluded.

`train_model` samples batches of 128 with replacement, trains for 1600 updates,
and checks validation every 100 updates. The same architecture, optimizer and
three seeds are used for both sizes. It selects the checkpoint using structural
validation loss only, then exports embeddings/scaler statistics from the common
2000-tree subset. Equal update count controls optimization effort but does not
prove convergence.

Online `run_one` uses the same 3×-first evaluator for every baseline. It constructs
all TTNs for parameter/padded screening, but only selected TTNs for random/BO.
GNN BO starts with SCENT, then calls the existing target-specific GP proposal
routine; its encoder remains frozen. The first incumbent with minimum CNOT count
is retained, and its depth is reported. The search is not depth constrained.

Five untouched overlapping instances (seeds 14–18) receive three repetitions,
with model seeds paired to search repetitions. Execution order cycles across
methods to balance positions. Methods are independently compiled without shared
outcome caches. Fixed candidate checkpoints are 1/2/4/8/16. Measured elapsed
costs are reported, but these capped runs are not an uncapped fixed-time benchmark.
Offline training is accounted separately, with untimed preprocessing disclosed.

`scripts/report_revised_gnn.py` verifies the corpus splits, source/model hashes,
strict accuracy requirements, attempt-start counts and incumbent histories, then
writes the [comparison report](artifacts/revised_gnn/REPORT.md). All targets remain
in the aggregate, including any where no method improves SCENT.
