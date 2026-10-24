# Learning-based TTN structure search — implementation plan

Updated: 2026-10-20. This is a research implementation plan, not a statement of results.

## Objective and scope

Given a classically accessible target state, a fidelity threshold and a classical search budget, return the lowest-CNOT verified feasible circuit found through TTN structure search. Optimize final squared fidelity F = |<psi|U|0>|² subject to F >= F0; report depth, runtime and failures separately. Keep physical bit labels fixed. Rank selection is controlled by a shared precision policy, not learned.

The first cycle uses fixed-size, nonnegative targets, frozen candidate pools and sequential constrained Bayesian optimization. It does not claim cross-size generalization. The structure encoder is self-supervised; the online performance surrogate learns from actual candidate evaluations.

## Execution order

1. Canonical labeled trees and small target states.
2. Dense hierarchical SVD → TTN → reference isometry circuits → exact verification.
3. Candidate headroom diagnostics and simple search (stage A).
4. Few-observation predictive diagnostics and random-pretrained GNN (stage B).
5. Mixed pretraining (stage C).
6. Factorial candidate/pretraining ablation (stage D).
7. Independent tests, TCI/research compiler, hardware and size extensions (stage E).

TCI and reproduction of the paper's approximate compiler must not block reference-backend experiments. Every backend needs its own validation and conclusions; success under the reference compiler does not establish success under SCENT's approximate compiler.

## Upstream SCENT

Pin treetensorcomp revision f05c28fbf959ecc023fd4ed89d1e00691d5ca564:
https://gitlab.com/gohmat/treetensorcomp

The inspected code provides QuanticsFunction, IndexID, TTNGraph, Clusterer, TreeTensorNetwork and Interpolator. Circuit synthesis was not found in the inspected release. Keep upstream source isolated and retain its AGPL-3.0-only metadata. Resolve its SciPy requirements in a separate compatible environment if necessary; do not assume the packaging constraints reproduce the example environment.

Keep the preparation root distinct from the backend's movable gauge center. Some upstream constructors suppress a degree-two root. The adapter must preserve labeled partitions and restore the preparation root explicitly. Treat graph reorientation as gauge bookkeeping, not as a new search candidate. Any adapted SCENT implementation must be identified and its affinity metric documented.

## Architecture and records

Use `src/ttn_search/` for trees, targets, reference/SCENT/TCI backends, verification, features, search, datasets, models and experiment orchestration. Keep configurations in `configs/`, correctness checks in `tests/`, and generated artifacts in `artifacts/`.

Core contracts:

- TargetSpec: target parameters, normalized amplitudes/function, grid, physical bit order and hash.
- TreeSpec: rooted labeled binary topology, canonical serialization and hash.
- PipelineConfig: construction, precision, synthesis, routing, resource and seed policies.
- BuildResult: normalized TTN, local isometries, rank profile, index maps and reconstruction evidence.
- CompileResult: circuit, physical wire map, ancilla policy and synthesis diagnostics.
- EvaluationRecord: status, all three fidelities, CNOT/depth, timings, configuration/software hashes and artifacts.
- SearchResult: best feasible evaluation or explicit no-feasible-result status, complete trajectory.

All methods call the same evaluator. Cache keys include target, rooted tree, pipeline, software/code version and evaluation seeds. Persist records atomically and support resumption. Separate actual evaluation time from cache lookup time.

## Stage A: reference pipeline and non-neural search

### Development configuration

- Fixed n=8; smaller states for correctness tests.
- Hierarchical and overlapping-correlation targets; initially three instances per family.
- Dense hierarchical SVD construction and exact isometry synthesis.
- Exact statevector verification; all-to-all connections; single-qubit rotations plus CX.
- Primary final fidelity 0.999.
- Diagnostic pools: 32 unique trees, optionally 64, evaluated completely.
- Search pools: 256 unique trees, budgets 16/32/64 including eight common initial evaluations.
- Three search seeds initially; measure evaluation cost before expanding.

Generate normalized amplitudes proportional to sqrt(p(x)) for probability encoding, or explicitly mark function-amplitude encoding. Fix finite grid, variable/bit mapping and domain. Build positive-definite covariances from factor loadings plus diagonal noise. Do not expose hidden generating groups to the model.

### Trees and circuits

Canonicalize sibling order and internal numbering while retaining leaf labels and preparation root. Validate connected acyclic binary trees and complete unique leaf coverage. Generate SCENT, balanced, comb, random-merge and legal local-reconnection candidates; deduplicate before evaluation. Random merging is not uniform sampling over all trees.

For the dense reference, recursively construct nested subspaces and local isometries, normalize the root, and reconstruct independently. Embed non-power-of-two bond spaces in qubit registers, complete unused columns isometrically, and explicitly account for all wires. Verify that circuit output matches the TTN and original state in a fixed physical bit order. Count gates after decomposition, not from tensor size. A layout relabeling is allowed on all-to-all hardware only when the full circuit is consistently mapped to fixed logical outputs; routing requires separate accounting.

### Evaluation policy

An evaluation is C(target, tree, precision policy, compiler budget, seed protocol), not an intrinsic cost of a tree. Freeze tolerance attempts, iteration limits, restart counts, seeds, time/memory limits and output-selection rules. Return the lowest-CX feasible circuit found by this policy, with deterministic tie handling. Charge all construction/compilation attempts. Keep cost and fidelity associated with the same circuit.

The first deterministic reference compiler can use a single synthesis attempt per construction tolerance; declare this explicitly. Approximate synthesis retries are a later extension. Timeouts and numerical failures are distinct from measured infeasibility and never get invented fidelity or cost labels.

Record F_repr, F_comp and F_final separately. Local SVD/TCI tolerances are not global fidelity guarantees. For E=1-F0, start with global representation/compilation error budgets 0.1E and 0.4E. Verify actual global errors. The bound is e_final <= (sqrt(e_repr)+sqrt(e_comp))². A predeclared robustness allocation is 0.2E/0.2E; rerun representative searches, not only winning trees. Exact synthesis may render this compilation allocation inactive; report that limitation.

### Baselines and initialization

Compare single SCENT, random search, raw relationship features + BO and PCA features + identical BO. Relationship features are upper-triangular LCA depths/n and leaf depths/n. Fit PCA on declared structural training data; any unlabeled pool preprocessing must be identified as transductive and shared across methods.

Every main search uses the same pool and eight initial points including the original SCENT tree. SCENT consumes one evaluation. Report total and additional evaluations beyond SCENT, and count tree-discovery overhead. The single-SCENT baseline does not artificially consume the search budget.

Use separate cost and fidelity models. After a feasible incumbent exists, score unseen candidates by expected cost improvement times feasibility probability. Before that, use feasibility and uncertainty exploration. Define fallback proposals when GP fitting lacks sufficient observations. Never encode operational failures as fidelity measurements. All attempted candidates consume budget; duplicates are excluded. Use independent deterministic seed streams.

Run a single trajectory to budget 64 and extract 16/32/64 checkpoints. A 64-tree fully evaluated pool only diagnoses headroom and discovery speed; its terminal optimum cannot distinguish methods. Frozen 256-tree pools need not be fully evaluated.

### Stage A deliverables and gate

Deliver a tested evaluator, reproducible tree–fidelity–CX–time records, SCENT/random/raw/PCA comparisons, and a report of headroom and feasibility. Initial smoke runs are engineering validation, not statistical evidence. Continue only if tree-dependent cost differences are credible and limited-budget search has useful room to improve.

## Stage B: test the representation hypothesis

Start with 2,000 unique random structures at fixed n. Direction-aware GIN: hidden dimension 64, four message layers, latent dimension eight, root/global readout. Inputs are node type, physical label, relative depth and subtree size. Reconstruct LCA and leaf depths with equal-weight MSE. Do not include target tensors, fitted ranks, source labels or circuit quality in encoder inputs.

Using complete small-pool development data, fit raw/PCA/GNN surrogates on shared subsets of 8 or 16 observations. Evaluate low-cost feasible candidate retrieval, feasibility, recommendation regret and feasible-only ranking on held-out candidates. Examine constraint calibration where data allow it. Then compare real BO trajectories. Reconstruction accuracy alone is insufficient.

Test internal-renumbering/sibling invariance, physical-label sensitivity, embedding collapse and long-comb behavior. If needed, replace shallow message passing with full upward/downward tree traversal. Do not expand to 10,000 trees without useful downstream evidence.

## Stage C: mixed structural pretraining (revised idea §6.3)

**Status: on hold.** Run the bounded decision experiment below before corpus scaling or mixed pretraining.

Target unique proportions are random/SCENT/SCENT-perturbation = 70/10/20. Pilot counts are 1,400/200/400; full counts 7,000/1,000/2,000. These are post-deduplication targets, not replication quotas. Bound generation effort; fill unresolved deficits with random structures and report actual proportions.

SCENT parents come from an isolated pretraining target library. Run only necessary affinity/tree discovery, not subsequent TCI or circuit evaluation. Account separately for expensive prerequisites. Descendants use 1–3 legal reconnections and a per-parent cap, preserve labels, and require no target refitting. SCENT provenance is not a quality label.

Split targets and related parameter variants before generating trees. Parents and all descendants share a lineage/split. Random ancestors and descendants follow the same rule. Globally eliminate cross-split canonical duplicates without breaking lineage assignments. Aim for 80/10/10 target-group splits; report actual tree counts. Use a common structural validation set excluded from both random-only and mixed training corpora.

Final state-preparation test targets never supply pretraining structures. Online SCENT on a test target is allowed but cannot feed back into training or tuning. Accidental topology overlap is not by itself test-target leakage; an unseen-topology claim requires its own hash-disjoint protocol.

Store tree hash/serialization, n, source, generator/seed, target/group IDs, parent hash, lineage, perturbation count, split, generation-run ID and SCENT-config hash. Record shared parent costs once in a generation-run table. Keep provenance for discarded duplicate attempts. Equal sample weights by default; source metadata stays outside the encoder. Match unique training counts, model, optimizer and update counts between corpora.

## Stage D: factorial ablation

Compare random-only versus mixed pretraining crossed with random-only versus mixed online pools. Within a pool condition share candidates and initial observations between encoders. Across pool conditions fix pool size and a replacement policy with a shared random component. The random-only pool condition does not force SCENT initialization. Count online SCENT time. Retain raw/PCA and random-search controls. Tune mixture only on development targets with a small predeclared comparison set.

## Stage E: independent validation and extensions

Freeze protocol before final testing. Expand to chain, hierarchical, overlapping and non-Gaussian target families, initially up to ten independent targets per family when resources allow. Aggregate search seeds within each target before paired bootstrap across targets. Final compiler validation gives every method the same independent seeds/opportunities; preserve original search observations and report all validation outcomes.

Add TCI, the paper's approximate compiler, n=12, hardware routing, dynamic pools and MPS comparisons only as separately validated extensions. Hardware runs count routing and SWAP decomposition. Larger systems require tractable exact verification or clearly labeled bounds/estimates; two compact TNs do not automatically have a cheap overlap contraction.

Report best feasible CX versus evaluations and actual time, success rates, paired differences, depth/fidelity tradeoffs and total costs. Include data generation, pretraining, affinity, proposal, evaluation/retries and final verification. Cached replay is a development tool, not evidence of online wall-clock speedup. Keep failed runs visible. Compute amortization only when measured savings justify it.

## Stop/redirect decisions

| Observation | Decision |
|---|---|
| Trees yield almost identical costs | Inspect targets, truncation and compiler; pause training |
| Random search already finds good trees quickly | Measure whether model overhead is justified; do not cherry-pick hard cases |
| Feature BO helps but GNN does not | Inspect representation/training objective before expanding data |
| Random pretraining helps, mixture does not | Keep random training and report the mixture ablation |
| Reference success does not transfer to research compiler | Explain backend dependence and narrow claims |
| Scaling fails | Separate representation, feasibility, runtime and compiler limitations |

## Validation priorities

Test canonical identity and leaf coverage; dense reconstruction and isometry direction; complex phases and non-power-of-two ranks; compiler bit order and padding; all three fidelity definitions; cache separation; failure/budget accounting; common initialization; lineage isolation and unique-source proportions. Avoid redundant tests of mere configuration values.

## Work estimate and status

A first research cycle may take 4–6 weeks if a usable compiler is available; reproducing approximate synthesis is separately estimated after its audit. The immediate deliverable is trustworthy small-system data and evidence of search headroom, followed by GNN and mixture validation. Execution status and measured results belong in artifact reports and README, not as assumed outcomes in this plan.

## Stage B execution protocol (2026-10-21)

The first representation pilot uses 2,000 random training trees and separate
250-tree validation/test splits, excluding all structures in the six original
candidate pools. Train three seeds for 100 epochs; select each checkpoint solely
by structural validation loss. Compare raw, PCA, frozen trained GNN, and an
untrained GNN, fitting their scalers to the same training corpus.

To remove the earlier heuristic-prefix bias, each downstream pool is SCENT plus
127 uniformly sampled candidates from its original 256-tree pool, chosen without
cost labels. Fully evaluate these pools for held-out prediction (8/16 known
labels, five splits) and replay search (common eight initial points, three seeds,
budgets 16/32/64). Random and one-hop-neighborhood-first search remain baselines.
This smaller-pool experiment is compared internally, not directly against Stage A
endpoints. All six targets are development targets. Training-seed repeats are not
independent target samples. Feasibility discrimination can only be assessed if
infeasible candidates occur. Do not advance to mixed pretraining based on
reconstruction loss alone. Scripts, exact protocol, training curves, checkpoints,
labels, and trace audit are part of the delivery.

Stage B pilot outcome: implemented and run with all three training seeds, 144
search trajectories, 360 prediction fits, and an independent audit of 9,216
search steps. Reconstruction improves over a constant-mean predictor, but the
learned embeddings do not consistently outperform simpler search baselines.
The gate to Stage C / 10,000-tree pretraining is **not met**. See
`artifacts/stage_b/REPORT.md` for per-target results, limitations and follow-up
recommendations. This is a negative development result for the tested setup,
not a general rejection of learning-based tree search.

## Next experiment: neighborhood validation on new instances

Freeze `configs/neighborhood_validation.json` before evaluation: n=8,
existing hierarchical/overlapping generators with unseen seeds 3–7, search
seeds 0–2, and 8/16/32 additional unique evaluations beyond SCENT. Use no
prebuilt candidate pool. Compare global random pair-merging trees, random-order
one-hop SCENT neighbors followed by global random fallback, and iterative
first-improvement neighborhood search. Iterative random fallback improvements
also change its incumbent; report this restart behavior explicitly. All methods
use the same reference evaluator and charge failures against their budgets.

This experiment validates new instances within known families. It is not a
larger-system or new-family test and does not establish novelty. Shared caches
reduce computation; report measured execution and charged evaluation times
separately without claiming independent online speedups. Preserve per-target
results and aggregate search seeds within each target.

For mechanism analysis, preselect the first successful improving neighbor and
first feasible non-improving neighbor from iterative search seed 0 per target.
Record removed/added leaf partitions, node ranks, parameter counts, tolerance
attempt, final fidelity/CNOT, and standalone local synthesis costs. Separate
local counts from the globally optimized circuit; their sum need not match.
After this validation, decide whether a bounded fixed-data GNN training extension
is still informative. No mixed-pretraining expansion is authorized by these
results alone.

Neighborhood validation completed on the ten declared new instances: 90
trajectories, 2,056 unique target–tree evaluations, and 2,880 additional search
steps. All hierarchical runs retain SCENT's 75 CNOT. At 32 additional evaluations,
initial-neighborhood search loses to random on all five overlapping targets.
Iterative search wins the mean (344.73 versus 354.13 CNOT), but wins on only one
target and loses on four. On that winning target, all three seeds reach 273
through local moves. A post-result mechanism case study examines the 9-to-8 rank
transition and its change from four to three padded bond-register qubits.
These results narrow the research signal to multi-step, instance-dependent
improvements; they do not establish a general neighborhood-search advantage.
Full evidence and limitations are in `artifacts/neighborhood_validation/REPORT.md`.

## Stage C expansion on hold: bounded decision experiment

Do not scale the pretraining corpus or start mixed random–SCENT pretraining yet.
Run `configs/decision_experiment.json` first, with three fixed components:

1. Reproduce the 504→279 transition on overlapping_4 and compare per-node local
   synthesis. Sweep local truncation tolerance by 0.8/0.9/1.0/1.1/1.2 while
   retaining the original global error requirements; use one attempt per setting
   to isolate sensitivity rather than hiding it behind retries.
2. Extract first-attempt, target-dependent rank/width/isometry-dimension
   descriptors across saved evaluations. Measure construction overhead. Primary
   held-out tests use the complete Stage B pools and SCENT plus global-random
   samples on the ten new instances. Exclude adaptive neighborhood samples from
   primary prediction metrics, retain them in the extraction audit, and report
   counterexamples as well as favorable cases. Compare fixed parameter/padded-size
   screening and a small GP against topology features and random selection.
3. Keep the original 2,000 training trees, validation/test splits, architecture,
   optimizer and seeds fixed; compare 100 with 300 epochs. Since saved checkpoints
   lack optimizer state, rerun from the same initial seeds and require the first
   100 epoch histories to reproduce before interpreting the extension. Select
   checkpoints only by reconstruction validation loss, then use identical Stage B
   held-out splits and search trajectories. No data-size or architecture tuning.

Descriptors are not free topology features: screening constructs every candidate
in the compared set, and that cost must be reported. Existing development labels
and adaptive-union data do not constitute independent test evidence. If extended
training improves only reconstruction, pause this pretraining recipe. If simple
rank/width screening works reliably, implement and validate that simpler method
before adding learning. Pool enlargement is deferred.

Decision-experiment outcome: the 504→279 cost transition survives the tested
±20% tolerance sweep. Two local blocks explain the 225-gate difference. Across
existing overlapping targets, first-build padded-isometry sizes achieve strong
within-target ranking and low recommendation regret at millisecond-scale dense
construction cost; the learned rank/width GP did not beat the fixed score on
retrieval. This supports validating simple screening before adding learning.

The 300-epoch GNN runs reproduce their original first 100 epochs exactly.
Reconstruction improves in all seeds; budget-64 search also improves in all three
seed means, so this is not a reconstruction-only result. Small-budget and
held-out retrieval improvements remain inconsistent, and the strongest existing
baseline is not beaten. Retain extended models as secondary baselines; keep
mixed pretraining and corpus scaling on hold. Full results, counterexamples,
construction accounting, and limitations: `artifacts/decision_experiment/REPORT.md`.

## Frozen screening benchmark after the decision experiment

Stage C scaling and mixed pretraining remain on hold. The authorized next test is
`configs/screening_benchmark.json`: ten untouched eight-qubit instances (seeds
8–12 in both existing families), three repetitions, and identical pools of SCENT
plus 127 unique random pair-merge trees. No neighborhood enrichment is inserted.

Compare SCENT alone, random order, SCENT-initialized random order, parameter-count
screening, padded-size screening, and all three frozen extended GNN models.
Fixed screening requires no circuit labels and no forced SCENT compilation.
Construct each screened TTN once and reuse it for its first compilation attempt;
retain the original fidelity checks and tolerance retries.

Run each method independently with no shared representation or outcome cache.
Report K=1/2/4/8/16 candidate budgets, actual synthesis-call budgets including
retries, and actual elapsed-time prefixes. Charge pool generation, whole-pool
screening, model setup, ranking/proposals, compilation, retries, verification and
artifact writing. Report common/offline setup separately. Methods stop at sixteen
candidates, so time curves are bounded-shortlist comparisons. Randomize execution
order; freeze models and all settings before obtaining new target labels.

Audit score ordering, identical pools, reuse accounting, fidelity and best-so-far
prefixes. The complete experiment/code walkthrough is `POST_STAGE_B_GUIDE.md`;
results and timings are in `artifacts/screening_benchmark/REPORT.md`. The next
scientific decision is whether the fixed screeners justify a modest larger-system
or different-compiler validation, not automatic resumption of mixed pretraining.

Screening outcome: all 180 runs completed and passed the artifact audit; 2,430
candidate evaluations required 2,495 compiler calls, including 65 retries.
All final fidelities meet 0.999 (minimum 0.999892416027). On untouched overlapping
targets, padded versus parameter screening gives mean CNOT 352.00 versus 456.20
at K=1, 345.93 versus 372.93 at K=4, and the same 339.80 at K=16. At two seconds,
including construction and verification, all runs have results: padded 347.33,
parameters 428.00, random 412.07, SCENT-initialized random 480.53, frozen GNN BO
495.67. The advantage is earlier useful selection, not a better final endpoint
than parameter screening. Hierarchical SCENT is already strong (75.8 versus
screeners' 75.2). Prioritize a modest generalization check; retain the Stage C hold.

## Revised Stage C: scaling and compiler dependence

The original larger/mixed GNN pretraining remains on hold. Revised C1 freezes the
padded-size rule and reference compiler, beginning with a 10-qubit resource pilot
(`configs/stage_c1.json`). Use two untouched targets (one per existing family,
seed 13), 128 candidates, and five baselines: SCENT, random, SCENT-initialized
random, parameter count, and padded size. This single-repetition pilot determines
feasibility; it is not sufficient for a statistical generalization claim.

Run fixed-evaluation checkpoints 1/2/4/8/16 separately from a 30-second fixed-time
experiment. Time runs may continue through the entire pool; no 16-query cap.
A parent process enforces the deadline and excludes unfinished results. Fixed-
evaluation runs have a 300-second resource safety ceiling; missing checkpoints
must remain missing. Record first verified result, completed prefixes, retries,
whole-pool preparation and peak resident memory. Interrupted attempt artifacts
are retained; completed-call counts do not include an interrupted in-flight call.
The fidelity requirement is 0.999, not 0.9999.

Review 10-qubit runtime/memory and selection results before deciding whether to
run 12 qubits. C2 follows on a small subset with a materially different synthesis
backend; changing seed or optimization level alone does not satisfy C2.

### C1 follow-up: pool headroom before further scaling

Complete the frozen overlapping seed-13 pool offline before expanding targets.
`scripts/diagnose_c1_pool.py` reuses the first existing completed record per tree
in a fixed path order, verifies duplicate outcome agreement, and evaluates missing
candidates with the unchanged precision policy and a 60-second candidate ceiling.
Report verified pool headroom, strict improvements/ties, score ranks and top-K
retrieval. Operational failures/timeouts leave uncertainty about the optimum.
This is an offline diagnostic, not online timing evidence. New synthesis starts
are persisted before entering the compiler; JSON writes already use atomic
replacement in the shared helper.

The subsequent independent validation should use overlapping seeds 14–18 at ten
qubits, three timing repetitions per method, frozen scores and generation policy,
and balanced method order. Integrate attempt-start logging into that formal
runner and report all interrupted calls. Do not treat timing repetitions as
independent targets. Twelve-qubit feasibility and C2 remain secondary until the
headroom result is interpreted. Expanded GNN pretraining remains on hold.

### Compilation/accuracy diagnostic after depth analysis

Completed a frozen three-tree, 21-configuration probe on overlapping seed 13:
SCENT, original padded-first candidate 74, and fixed pool candidate 1. Reference
local tolerance multipliers were 1/3/10/30/100; full-unitary completion plus
Quantum Shannon decomposition was tested at 1/10. Final fidelity 0.999 determines
diagnostic feasibility; original sub-budget feasibility is separately recorded.
This changes the local tolerance policy, not the historical benchmark results.

SCENT at multiplier 3 yields 592 CNOTs and two-qubit depth 436, versus 1163/900
nominally, with final fidelity 0.99996344 and the original global sub-budgets still
satisfied. Factor 100 fails final fidelity for all three trees. Critical-chain
analysis identifies large root and six-leaf isometries; their rank/register changes
account for the observed depth reduction. Full-unitary QSD completion is much
worse than the reference path on these examples, without ruling out other synthesis
methods. See `artifacts/compilation_probe/REPORT.md`.

Next validate and freeze a shared tolerance policy before new-target comparisons.
Rebuild target-dependent descriptors under it, and give SCENT and every selector
the same policy. These development observations do not justify a universal factor,
a new hardware depth requirement, or expanded GNN training yet.

## Validated 3×-first policy and bounded GNN study

The six-instance development cohort covers both families, eight qubits (seeds
0/3), and ten qubits (seed 13); SCENT/balanced/fixed-random trees are paired at
1× and 3×. Of 36 direct evaluations, 35 pass all global requirements. One 3×
hierarchical random case exhibits a large compilation error despite acceptable
representation error; explicit policy replay rejects it and succeeds at 1×.

Freeze local factors 3→1→0.1→0 with first-feasible acceptance. Retain representation
infidelity ≤0.0001, compilation infidelity ≤0.0004, and final F≥0.999. All failed
attempts and retries count. Rebuild all 128 overlapping seed-13 candidates under
this policy and use first-3× descriptors: all pass, five fall back to 1×, and SCENT
remains the sole CNOT/depth frontier point at 592/436. See
`artifacts/revised_policy/REPORT.md`.

The authorized GNN expansion is now a bounded random-corpus ablation, not mixed
pretraining: nested 2000/10000-tree corpora; unchanged ten-qubit architecture;
1600 optimizer updates with batch size 128, Adam .001, validation every 100
updates; three paired seeds. A common 500-tree validation set selects checkpoints;
a separate 500-tree test set measures reconstruction. Training excludes the frozen
benchmark pool topologies. Equal updates do not establish convergence.

Untouched overlapping seeds 14–18 have identical 128-tree pools across seven
methods: SCENT, random, SCENT-initialized random, parameter screening, padded
screening, and both GNN corpus sizes with BO. Three repetitions pair model seed
and search seed. Method positions use cyclic balanced orders. Fixed-evaluation
checkpoints 1/2/4/8/16 are reported with actual online elapsed cost and incumbent
depth. This study does not claim an uncapped fixed-time comparison or depth-aware
BO. Offline model training costs are reported separately. The evaluator remains
fixed throughout; mixed pretraining and twelve-qubit expansion are not included.

Bounded GNN study outcome: all 105 runs completed (1455 candidates, 1485 synthesis
calls, 32 retry attempts; all final evaluations feasible). The 2000- and 10000-tree
models tie at every checkpoint in all 15 paired runs, retaining SCENT throughout.
Mean final CNOT is 663.8 for SCENT and either GNN, 653.8 for SCENT-initialized
random, and 633.8 for either screener. Padded screening reaches that mean at K=1;
parameter screening reaches it by K=4. Target 15 supplies the observed improvement
955→805 CNOT and depth 795→742; no method improves SCENT on the other four targets.
No exhaustive-optimum claim is made for these new pools. All model checkpoints
are selected at the final update, so convergence remains unresolved. Do not
interpret this fixed-update null result as a general impossibility result for GNNs.
See `artifacts/revised_gnn/REPORT.md` for paired results, costs and limitations.

### Follow-up decision checks: compiler and ranking audit

- Localized the hierarchical seed-13 3× failure to one Qiskit UCGate inside a
  64×4 isometry. Independent dense completion validates the TTN and physical
  register mapping; replacing only that block restores fidelity. Preserve the
  standalone reproducer and strict expected-failure regression tests. The exact
  internal numerical defect and an efficient production repair remain open.
- Audited 617 unique candidate/target evaluations across six ten-qubit
  overlapping targets. Only seed 13 is exhaustive. Padded screening's first
  candidate matches the best observed CNOT on every target; other cost-ranking
  errors do not yet demonstrate useful shortlist headroom.
- Existing evaluation labels remain specific to the frozen compiler plus
  failure/fallback policy. Do not mix diagnostic replacement costs into them.
  Version and validate any compiler repair before refreshing labels.
- Do not automatically launch supervised GNN ranking. First establish additional
  shortlist headroom or a measured descriptor-computation bottleneck. The
  audit reports retain unknown pool optima explicitly.

### Compiler repair implemented (separate version)

- Added `verified-uc-diagonal-v1-qiskit-2.6.0` as an explicit alternative to the
  frozen reference compiler. It independently verifies multiplexer branch
  matrices, corrects erroneous diagonal phases, and uses verified local Euler
  synthesis when the discrepancy cannot be corrected diagonally.
- The bounded paired development validation covers 28 cases: the original
  hierarchical failure, all 14 unique historical retry candidates from the
  six-pool audit, and 13 successful controls. Repaired compilation passes all;
  12 frozen direct compilations fail. Successful controls retain CNOT and depth.
- Three direct 3× cases still violate the representation sub-budget. The common
  tolerance fallback remains required. The repair is not a relaxation of
  accuracy requirements.
- Production benchmark runners and historical labels are unchanged. Next:
  explicitly integrate this version into a new evaluator run, refresh comparison
  labels under one common policy, then complete the remaining candidate pools.
  No new GNN training is authorized by these repair results alone.

### Separately versioned evaluator refresh

Implemented an isolated `verified_evaluator_v1` run with unchanged strict budgets
and retry ordering, explicit compiler/evaluator fingerprints, persistent call
accounting, and overwrite protection. The run repeats all 105 online method
comparisons using frozen checkpoints, then completes six 128-tree offline pools
using only repaired-version labels. Training remains on hold. Results are not
complete until `artifacts/verified_evaluator_v1/completion.json` and its audited
`REPORT.md` exist; progress is recorded in `status.json`.

### Repaired-evaluator completion and next bounded check

The separately versioned run completed: 105 online runs, 768 exhaustively
measured pool candidates, no failed final evaluations. All 105 final CNOT/depth
endpoints match the historical results. Padded screening selects the unique
pool Pareto candidate first in all six pools; SCENT is optimal in five, with
only overlapping seed 15 improving to 805 CNOT / depth 742. These are fixed-pool
optima, not global state-preparation bounds.

Completed a separate exact-subtree reuse probe at 8/10/12 qubits, both existing
families, seed 19, 128 candidates and three timing repetitions. All 2,304 paired
builds preserve amplitudes, ranks, scores and feasibility decisions. Median
construction savings are small, and twelve-qubit timing variability prevents a
strong speedup claim. The experimental cache is not integrated into evaluators.
See `artifacts/shared_svd_feasibility/REPORT.md` and `DECISION.md` for results and
memory/retention limitations. Continue to pause GNN training; benchmark breadth
and candidate-generation headroom are the next research questions.

### Interpretation correction and problem-suitability protocol

Small subtree-cache savings do not bound potential learned-prescreening gains.
The 12-qubit construction/reconstruction/verification/scoring total remains
about 4–5 seconds per 128 candidates in this probe; component costs were not
isolated. A learned descriptor substitute has not demonstrated end-to-end
benefit, but the cache result alone cannot establish its upper bound.

The next development experiment is specified in `PROBLEM_SUITABILITY_PROTOCOL.md`.
Keep ten qubits, the repaired compiler and strict policy fixed. Compare two
equal-size candidate generators across existing controls and a controlled
TTN-generated family. Exhaustively evaluate the bounded pools; retain every
instance and operational failure. Freeze instantiated targets/pools before
obtaining circuit labels. This is a protocol, not a completed or launched run;
GNN expansion remains on hold.

### 独立三叉预实验的顺序

保留问题适配性协议主体，优先执行二叉目标/候选多样性实验。随后按
`TERNARY_PREFLIGHT_PROTOCOL.md` 开展最多 16 个局部案例的同态分块实验：
每案例一个三叉块、三种二叉细化，另允许原始规范诊断，总计最多 80 次主/诊断
编译。选择目标和合并边不依赖线路输赢。先验证合并/细化保持同一近似态及
寄存器映射；新适配层独立版本化。新叉数的独立构建和 GNN 暂不启动。

### Problem-suitability implementation

Implemented controlled TTN targets, quota-checked random/mixed pools, an
instrumented SVD reference-equivalence check, fixed target-dependent feature
sets, versioned exhaustive evaluation, and a grouped diagnostic/report runner.
Correctness tests cover reproducibility, generating-cut rank bounds, independent
ZZ evolution, pool quotas and profile equivalence. Instantiation/fingerprinting
precedes compilation. Execution state lives in `artifacts/problem_suitability_v1`;
completion must be checked there rather than inferred from implementation status.
No GNN training or ternary experiment is included in this run.
